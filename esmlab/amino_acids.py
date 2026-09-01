"""Shared amino-acid domain data used across analysis, IO, and reporting."""

# The 20 canonical amino acids, in a fixed order used as the column order for
# every (sequence_length, 20) analysis array.
VALID_AMINO_ACIDS = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)

# Isoelectric points, used only to order the report matrix's rows by charge
# https://www.vanderbilt.edu/AnS/Chemistry/Rizzo/stuff/AA/AminoAcids.html
AA_TO_ISOELECTRIC_POINT: dict[str, float] = {
    "A": 6.11,
    "R": 10.76,
    "N": 5.43,
    "D": 2.98,
    "C": 5.15,
    "E": 3.08,
    "Q": 5.65,
    "G": 6.06,
    "H": 7.64,
    "I": 6.04,
    "L": 6.04,
    "K": 9.47,
    "M": 5.71,
    "F": 5.76,
    "P": 6.30,
    "S": 5.70,
    "T": 5.60,
    "W": 5.88,
    "Y": 5.63,
    "V": 6.02,
}
