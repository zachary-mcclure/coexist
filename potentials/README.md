# Interatomic potentials (NIST Interatomic Potentials Repository)

Downloaded 2026-07-02 from https://www.ctcms.nist.gov/potentials — see each
entry page for full provenance. Ladder rungs refer to
`docs/REVIEW_CHECKPOINT_10.md` §3.

| File | System | Ladder rung | Source entry | Citation |
|---|---|---|---|---|
| `CuNi.eam.alloy` | Cu-Ni | T1: isomorphous lens | [2019--Fischer-F-Schmitz-G-Eich-S-M--Cu-Ni](https://www.ctcms.nist.gov/potentials/entry/2019--Fischer-F-Schmitz-G-Eich-S-M--Cu-Ni/) | Fischer, Schmitz & Eich, *Acta Mater.* 176 (2019) 220–231 — fitted with attention to the Cu-Ni miscibility gap |
| `CuAg.eam.alloy` | Cu-Ag | T1: simple eutectic (flagship) | [2006--Williams-P-L-Mishin-Y-Hamilton-J-C--Cu-Ag](https://www.ctcms.nist.gov/potentials/entry/2006--Williams-P-L-Mishin-Y-Hamilton-J-C--Cu-Ag/) | Williams, Mishin & Hamilton, *MSMSE* 14 (2006) 817–833 — reproduces the Ag-Cu eutectic in the original paper |
| `NiAl.eam.alloy` | Ni-Al | T4: B2/L1₂ compounds, peritectics | [2009--Purja-Pun-G-P-Mishin-Y--Ni-Al](https://www.ctcms.nist.gov/potentials/entry/2009--Purja-Pun-G-P-Mishin-Y--Ni-Al/) | Purja Pun & Mishin, *Phil. Mag.* 89 (2009) 3245 |
| `AlPb.eam.alloy` | Al-Pb | T5: monotectic | [2000--Landa-A-Wynblatt-P-Siegel-D-J-et-al--Al-Pb](https://www.ctcms.nist.gov/potentials/entry/2000--Landa-A-Wynblatt-P-Siegel-D-J-et-al--Al-Pb/) | Landa et al., *Acta Mater.* 48 (2000) 1753 — glue-type, fitted for Al-Pb demixing |
| `PbSn.Memphis.meam` + `library.PbSn.Memphis.meam` | Pb-Sn | T2: two-lattice eutectic | [2018--Etesami-S-A-Baskes-M-I-Laradji-M-Asadi-E--Pb-Sn](https://www.ctcms.nist.gov/potentials/entry/2018--Etesami-S-A-Baskes-M-I-Laradji-M-Asadi-E--Pb-Sn/) | Etesami, Baskes, Laradji & Asadi, *Acta Mater.* 161 (2018) 320 — MEAM fitted to melting properties |
| `ZrNb5.eam.alloy` | Zr-Nb | all-solid monotectoid (first liquid-free invariant) | [2024--Fan-Z-Maras-E-Cottura-M-et-al--Zr-Nb](https://www.ctcms.nist.gov/potentials/entry/2024--Fan-Z-Maras-E-Cottura-M-et-al--Zr-Nb/2024--Fan-Z--Zr-Nb--LAMMPS--ipr1.html) | Fan, Maras, Cottura, Marinica & Clouet, *Phys. Rev. Materials* 8 (2024) 113601 — purpose-built for bcc-Nb/hcp-Zr precipitate coherency, i.e. exactly this system's interface physics |

Not yet sourced: Cu-Au EAM (Gola & Pastewka 2018 entry hosts files externally);
Al-Si (MEAM only on NIST — MACE-MP covers this rung instead).

Usage: `coexist.adapters.lammps_factory` builds LAMMPSlib calculator
factories from these files by element pair.
