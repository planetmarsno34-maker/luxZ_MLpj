from Bio import Entrez, SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os
import json

Entrez.email = "planetmarsno34@gmail.com"

ACCESSION = "CP000021.2"
UPSTREAM_BP = 300

LUX_LOCUS_TAGS = {
    "VF_A0593": "luxT",
    "VF_A0918": "luxG",
    "VF_A0919": "luxE",
    "VF_A0920": "luxB",
    "VF_A0921": "luxA",
    "VF_A0922": "luxD",
    "VF_A0923": "luxC",
    "VF_A0924": "luxI",
    "VF_A0925": "luxR",
    "VF_A1058": "qsrP",
}

# Operon block: luxC through luxG (co-transcribed unit)
OPERON_START_TAG = "VF_A0923"  # luxC
OPERON_END_TAG   = "VF_A0918"  # luxG

GB_CACHE = "data/CP000021.2.gb"


def download_genbank():
    if os.path.exists(GB_CACHE):
        print(f"Using cached GenBank file: {GB_CACHE}")
        return

    print(f"Downloading {ACCESSION} from NCBI (this may take a minute)...")
    handle = Entrez.efetch(db="nuccore", id=ACCESSION, rettype="gb", retmode="text")
    with open(GB_CACHE, "w") as f:
        f.write(handle.read())
    handle.close()
    print("Download complete.")


def extract_sequences(record):
    cds_records = []
    upstream_records = []
    chrom_seq = record.seq
    chrom_len = len(chrom_seq)

    operon_start = None
    operon_end = None

    for feature in record.features:
        if feature.type != "CDS":
            continue

        locus_tag = feature.qualifiers.get("locus_tag", [None])[0]
        if locus_tag not in LUX_LOCUS_TAGS:
            continue

        gene_name = LUX_LOCUS_TAGS[locus_tag]
        strand = feature.location.strand
        start = int(feature.location.start)  # 0-based
        end = int(feature.location.end)

        # Track operon boundaries
        if locus_tag == OPERON_START_TAG:
            operon_start = start
        if locus_tag == OPERON_END_TAG:
            operon_end = end

        # CDS sequence (coding DNA, same strand as gene)
        cds_seq = feature.extract(chrom_seq)
        cds_records.append(SeqRecord(
            cds_seq,
            id=locus_tag,
            description=f"{gene_name} CDS | {ACCESSION} | strand={'+' if strand == 1 else '-'} | {start+1}..{end}"
        ))

        # Upstream promoter region
        if strand == 1:
            up_start = max(0, start - UPSTREAM_BP)
            up_seq = chrom_seq[up_start:start]
        else:
            up_end = min(chrom_len, end + UPSTREAM_BP)
            up_seq = chrom_seq[end:up_end].reverse_complement()

        upstream_records.append(SeqRecord(
            up_seq,
            id=locus_tag,
            description=f"{gene_name} upstream_{UPSTREAM_BP}bp | {ACCESSION}"
        ))

        print(f"  {locus_tag} ({gene_name}): CDS={len(cds_seq)} bp, upstream={len(up_seq)} bp")

    return cds_records, upstream_records, operon_start, operon_end


def extract_operon(record, operon_start, operon_end):
    if operon_start is None or operon_end is None:
        print("Warning: could not find operon boundaries.")
        return None

    operon_seq = record.seq[operon_start:operon_end]
    return SeqRecord(
        operon_seq,
        id="lux_operon",
        description=f"luxCDABEG operon block | {ACCESSION} | {operon_start+1}..{operon_end}"
    )


def main():
    os.makedirs("data/dna", exist_ok=True)

    download_genbank()

    print(f"\nParsing GenBank record...")
    record = SeqIO.read(GB_CACHE, "genbank")
    print(f"Chromosome length: {len(record.seq):,} bp")

    print(f"\nExtracting lux gene sequences...")
    cds_records, upstream_records, op_start, op_end = extract_sequences(record)

    # Save CDS sequences
    cds_path = "data/dna/lux_cds.fasta"
    SeqIO.write(cds_records, cds_path, "fasta")
    print(f"\nSaved {len(cds_records)} CDS sequences → {cds_path}")

    # Save upstream/promoter sequences
    up_path = "data/dna/lux_upstream_300bp.fasta"
    SeqIO.write(upstream_records, up_path, "fasta")
    print(f"Saved {len(upstream_records)} upstream regions → {up_path}")

    # Save operon block
    operon = extract_operon(record, op_start, op_end)
    if operon:
        op_path = "data/dna/lux_operon_block.fasta"
        SeqIO.write([operon], op_path, "fasta")
        print(f"Saved lux operon block ({len(operon.seq):,} bp) → {op_path}")

    # Save metadata
    meta = {tag: gene for tag, gene in LUX_LOCUS_TAGS.items()}
    with open("data/dna/lux_gene_map.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved gene map → data/dna/lux_gene_map.json")

    print("\nDone. Files saved to data/dna/")


if __name__ == "__main__":
    main()
