import torch

# --------------------------------------
def get_weight_table():
    atomic_weights = torch.tensor([
        float('nan'),   # 0 - placeholder
        1.008,          # 1  H
        4.002602,       # 2  He
        6.94,           # 3  Li
        9.0121831,      # 4  Be
        10.81,          # 5  B
        12.011,         # 6  C
        14.007,         # 7  N
        15.999,         # 8  O
        18.998403163,   # 9  F
        20.1797,        # 10 Ne
        22.98976928,    # 11 Na
        24.305,         # 12 Mg
        26.9815385,     # 13 Al
        28.085,         # 14 Si
        30.973761998,   # 15 P
        32.06,          # 16 S
        35.45,          # 17 Cl
        39.948,         # 18 Ar
        39.0983,        # 19 K
        40.078,         # 20 Ca
        44.955908,      # 21 Sc
        47.867,         # 22 Ti
        50.9415,        # 23 V
        51.9961,        # 24 Cr
        54.938044,      # 25 Mn
        55.845,         # 26 Fe
        58.933194,      # 27 Co
        58.6934,        # 28 Ni
        63.546,         # 29 Cu
        65.38,          # 30 Zn
        69.723,         # 31 Ga
        72.63,          # 32 Ge
        74.921595,      # 33 As
        78.971,         # 34 Se
        79.904,         # 35 Br
        83.798,         # 36 Kr
        85.4678,        # 37 Rb
        87.62,          # 38 Sr
        88.90584,       # 39 Y
        91.224,         # 40 Zr
        92.90637,       # 41 Nb
        95.95,          # 42 Mo
        98.0,           # 43 Tc (most stable isotope)
        101.07,         # 44 Ru
        102.90550,      # 45 Rh
        106.42,         # 46 Pd
        107.8682,       # 47 Ag
        112.414,        # 48 Cd
        114.818,        # 49 In
        118.710,        # 50 Sn
        121.760,        # 51 Sb
        127.60,         # 52 Te
        126.90447,      # 53 I
        131.293,        # 54 Xe
        132.90545196,   # 55 Cs
        137.327,        # 56 Ba
        138.90547,      # 57 La
        140.116,        # 58 Ce
        140.90766,      # 59 Pr
        144.242,        # 60 Nd
        145.0,          # 61 Pm
        150.36,         # 62 Sm
        151.964,        # 63 Eu
        157.25,         # 64 Gd
        158.92535,      # 65 Tb
        162.500,        # 66 Dy
        164.93033,      # 67 Ho
        167.259,        # 68 Er
        168.93422,      # 69 Tm
        173.045,        # 70 Yb
        174.9668,       # 71 Lu
        178.49,         # 72 Hf
        180.94788,      # 73 Ta
        183.84,         # 74 W
        186.207,        # 75 Re
        190.23,         # 76 Os
        192.217,        # 77 Ir
        195.084,        # 78 Pt
        196.966569,     # 79 Au
        200.592,        # 80 Hg
        204.38,         # 81 Tl
        207.2,          # 82 Pb
        208.98040,      # 83 Bi
        209.0,          # 84 Po
        210.0,          # 85 At
        222.0,          # 86 Rn
        223.0,          # 87 Fr
        226.0,          # 88 Ra
        227.0,          # 89 Ac
        232.0377,       # 90 Th
        231.03588,      # 91 Pa
        238.02891,      # 92 U
        237.0,          # 93 Np
        244.0,          # 94 Pu
        243.0,          # 95 Am
        247.0,          # 96 Cm
        247.0,          # 97 Bk
        251.0,          # 98 Cf
        252.0,          # 99 Es
        257.0,          # 100 Fm
        258.0,          # 101 Md
        259.0,          # 102 No
        262.0,          # 103 Lr
        267.0,          # 104 Rf
        270.0,          # 105 Db
        271.0,          # 106 Sg
        270.0,          # 107 Bh
        277.0,          # 108 Hs
        276.0,          # 109 Mt
        281.0,          # 110 Ds
        282.0,          # 111 Rg
        285.0,          # 112 Cn
        286.0,          # 113 Nh
        289.0,          # 114 Fl
        290.0,          # 115 Mc
        293.0,          # 116 Lv
        294.0,          # 117 Ts
        294.0           # 118 Og
    ], dtype=torch.float32)

    return atomic_weights

import torch

def get_atomic_number_map():
    """
    Creates a lookup tensor to map (period, group) to atomic number.
    Indexed by chemistry convention:
        - Periods: 1 to 7 (rows 1 to 7)
        - Groups:  1 to 18 (columns 1 to 18)
    Entries without elements are NaN.
    This does not include the rare Earth elements. 

    Returns:
        torch.Tensor: A 2D tensor where map[period, group] gives the atomic number.
    """
    # Allocate with NaNs and 1-based indexing (add an empty 0th row and column)
    atomic_map = torch.full((8, 19), float('nan'), dtype=torch.float)#, float('nan'), dtype=torch.float)

    # === Period 1 ===
    atomic_map[1, 1]  = 1   # H
    atomic_map[1, 18] = 2   # He

    # === Period 2 ===
    atomic_map[2, 1]  = 3   # Li
    atomic_map[2, 2]  = 4   # Be
    atomic_map[2, 13] = 5   # B
    atomic_map[2, 14] = 6   # C
    atomic_map[2, 15] = 7   # N
    atomic_map[2, 16] = 8   # O
    atomic_map[2, 17] = 9   # F
    atomic_map[2, 18] = 10  # Ne

    # === Period 3 ===
    atomic_map[3, 1]  = 11  # Na
    atomic_map[3, 2]  = 12  # Mg
    atomic_map[3, 13] = 13  # Al
    atomic_map[3, 14] = 14  # Si
    atomic_map[3, 15] = 15  # P
    atomic_map[3, 16] = 16  # S
    atomic_map[3, 17] = 17  # Cl
    atomic_map[3, 18] = 18  # Ar

    # === Period 4 ===
    atomic_map[4, 1]  = 19  # K
    atomic_map[4, 2]  = 20  # Ca
    atomic_map[4, 3]  = 21  # Sc
    atomic_map[4, 4]  = 22  # Ti
    atomic_map[4, 5]  = 23  # V
    atomic_map[4, 6]  = 24  # Cr
    atomic_map[4, 7]  = 25  # Mn
    atomic_map[4, 8]  = 26  # Fe
    atomic_map[4, 9]  = 27  # Co
    atomic_map[4, 10] = 28  # Ni
    atomic_map[4, 11] = 29  # Cu
    atomic_map[4, 12] = 30  # Zn
    atomic_map[4, 13] = 31  # Ga
    atomic_map[4, 14] = 32  # Ge
    atomic_map[4, 15] = 33  # As
    atomic_map[4, 16] = 34  # Se
    atomic_map[4, 17] = 35  # Br
    atomic_map[4, 18] = 36  # Kr

    # === Period 5 ===
    atomic_map[5, 1]  = 37  # Rb
    atomic_map[5, 2]  = 38  # Sr
    atomic_map[5, 3]  = 39  # Y
    atomic_map[5, 4]  = 40  # Zr
    atomic_map[5, 5]  = 41  # Nb
    atomic_map[5, 6]  = 42  # Mo
    atomic_map[5, 7]  = 43  # Tc
    atomic_map[5, 8]  = 44  # Ru
    atomic_map[5, 9]  = 45  # Rh
    atomic_map[5, 10] = 46  # Pd
    atomic_map[5, 11] = 47  # Ag
    atomic_map[5, 12] = 48  # Cd
    atomic_map[5, 13] = 49  # In
    atomic_map[5, 14] = 50  # Sn
    atomic_map[5, 15] = 51  # Sb
    atomic_map[5, 16] = 52  # Te
    atomic_map[5, 17] = 53  # I
    atomic_map[5, 18] = 54  # Xe

    # === Period 6 ===
    atomic_map[6, 1]  = 55  # Cs
    atomic_map[6, 2]  = 56  # Ba
    atomic_map[6, 3]  = 57  # La
    atomic_map[6, 4]  = 72  # Hf
    atomic_map[6, 5]  = 73  # Ta
    atomic_map[6, 6]  = 74  # W
    atomic_map[6, 7]  = 75  # Re
    atomic_map[6, 8]  = 76  # Os
    atomic_map[6, 9]  = 77  # Ir
    atomic_map[6, 10] = 78  # Pt
    atomic_map[6, 11] = 79  # Au
    atomic_map[6, 12] = 80  # Hg
    atomic_map[6, 13] = 81  # Tl
    atomic_map[6, 14] = 82  # Pb
    atomic_map[6, 15] = 83  # Bi
    atomic_map[6, 16] = 84  # Po
    atomic_map[6, 17] = 85  # At
    atomic_map[6, 18] = 86  # Rn

    # === Period 7 ===
    atomic_map[7, 1]  = 87  # Fr
    atomic_map[7, 2]  = 88  # Ra
    atomic_map[7, 3]  = 89  # Ac
    atomic_map[7, 4]  = 104 # Rf
    atomic_map[7, 5]  = 105 # Db
    atomic_map[7, 6]  = 106 # Sg
    atomic_map[7, 7]  = 107 # Bh
    atomic_map[7, 8]  = 108 # Hs
    atomic_map[7, 9]  = 109 # Mt
    atomic_map[7, 10] = 110 # Ds
    atomic_map[7, 11] = 111 # Rg
    atomic_map[7, 12] = 112 # Cn
    atomic_map[7, 13] = 113 # Nh
    atomic_map[7, 14] = 114 # Fl
    atomic_map[7, 15] = 115 # Mc
    atomic_map[7, 16] = 116 # Lv
    atomic_map[7, 17] = 117 # Ts
    atomic_map[7, 18] = 118 # Og

    return atomic_map
