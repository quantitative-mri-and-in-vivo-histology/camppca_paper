"""Download the in vivo panel: 28 configurations, two subjects each.

OpenNeuro: each scan's DWI series with bval, bvec and, where the release has one, the JSON
sidecar, checked against the SHA256 the release records.

HCP (hcp01, hcp02): each series' NIfTI, bval and bvec from the hcp-openaccess S3 bucket, checked
against SHA256 values pinned here. Needs the aws CLI and HCP credentials in the environment.

    python scripts/datasets/panel/fetch.py [--skip-hcp]
"""

import argparse
import subprocess

from camppca_paper.openneuro import checksum, clone, fetch
from camppca_paper.panel import HCP, SCANS

HCP_BUCKET = "s3://hcp-openaccess/HCP_1200"
HCP_SHA256 = {
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir95_LR.nii.gz":
        "d40d1e831dbd9f1b1097e9e63b41023d56ef3a502a138f84cd7a958b042db66a",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir95_LR.bval":
        "0bc439ed687e97be746fd88815c072c3762873317f29567f5f17f38f618981db",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir95_LR.bvec":
        "c521fb0666b5d8d21cc47d7e34688a9a077feb15e6ece220cb869de5307090e1",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir96_LR.nii.gz":
        "9b43e24450a03e9f8cb2dba0c0cca19b71471625b51c71109af1112a54804ad2",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir96_LR.bval":
        "eca9707a39b699f4b94ec9d17642bbb4b2601e797de0aca1aae42f47538ae30c",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir96_LR.bvec":
        "af81417ed465ee0e138477abadd143d3e47632f755782e3ef4bd6b1414090a43",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir97_LR.nii.gz":
        "d1393fc48a10b4ac339884dcdc3eb0c59976149dda740fad285609f03c611f4f",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir97_LR.bval":
        "03ca9527772854602e114160b691b9b27a99dc1a10ee973b70cb9e20efb8f17f",
    "100206/unprocessed/3T/Diffusion/100206_3T_DWI_dir97_LR.bvec":
        "c044859849b9334d295d1a5dc9333e5250b72db695a6ad0064a16276df02ef0c",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir95_LR.nii.gz":
        "665d4c45a1783ecdde747b81c276d9ef572ec8f9964ce7e56e9083adc133c760",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir95_LR.bval":
        "89adbf2520e9ff2bf73f9d1783aaef31560a205f3890aabbbbd654002755a475",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir95_LR.bvec":
        "fe630f69ea92eef080b3352773684bca772c27a927b34ca6637602f40ac3693d",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir96_LR.nii.gz":
        "3bf47cf49e565bcbd5b19bbf9b1df5b8f26bda488629f59e36f246e8a63ce43c",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir96_LR.bval":
        "d577f2a9d66c6ab213bd5f85fa4a13e9164111306adbcba17878125e97d57b27",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir96_LR.bvec":
        "74333dd7370e6b50430d156a861551108d54d0f8deab6d362d0634fcd4f74b4e",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir97_LR.nii.gz":
        "9f60934f00478347e0a27b63adf132f25e5fc974b565298a86887a8da1522ea3",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir97_LR.bval":
        "8180208dfb61b44bd4f580eafa6b791e75a99e47dab72116c3e7673783060599",
    "103818/unprocessed/3T/Diffusion/103818_3T_DWI_dir97_LR.bvec":
        "cc836c06312c3eb9f534705388b7a529e87ea2ff2182634e3cedd96b00081e36",
    "104416/unprocessed/7T/Diffusion/104416_7T_DWI_dir71_AP.nii.gz":
        "5d4fd1669d521a105318158368876e878b586cd4aecc41ecd3e04d4f643e51af",
    "104416/unprocessed/7T/Diffusion/104416_7T_DWI_dir71_AP.bval":
        "bb4808fc6d0869e2db3d1fafb9dff86495b6ffccd900e27edc0d0475296bf2c9",
    "104416/unprocessed/7T/Diffusion/104416_7T_DWI_dir71_AP.bvec":
        "c8b7502e4de5e4fc7f62b6b5a9b8c420173c2115646e62ff1d7422b2c121a3e3",
    "104416/unprocessed/7T/Diffusion/104416_7T_DWI_dir72_AP.nii.gz":
        "05a6fcfdcf09a0399486b179989f16382e1263851cf8e1458bbe51a120ce7bf4",
    "104416/unprocessed/7T/Diffusion/104416_7T_DWI_dir72_AP.bval":
        "dab666295f49c88ee7833dd366ef45f2708e308ef7b00679398172e23f9baf6a",
    "104416/unprocessed/7T/Diffusion/104416_7T_DWI_dir72_AP.bvec":
        "fa1bda2ea700a392ef9a9433bbb5bde59fd3e1faee07a8220569caf80aaa5d89",
    "114823/unprocessed/7T/Diffusion/114823_7T_DWI_dir71_AP.nii.gz":
        "a19729c285e0f9a288213cd63c854034d1f9b4879baf0bcd877bdedcb82394e9",
    "114823/unprocessed/7T/Diffusion/114823_7T_DWI_dir71_AP.bval":
        "6d4858331ba14c48f98565df10a8e431b5c4570be500a4d01c50089df367ecdf",
    "114823/unprocessed/7T/Diffusion/114823_7T_DWI_dir71_AP.bvec":
        "4819d36694ee7781969299188a7ca7d2dcd589f157910d404d54c5171e5377ca",
    "114823/unprocessed/7T/Diffusion/114823_7T_DWI_dir72_AP.nii.gz":
        "775dc907cb5a721e49c12f5098cb0eaf56b5e02ab34fe0d638912ba37f3f6745",
    "114823/unprocessed/7T/Diffusion/114823_7T_DWI_dir72_AP.bval":
        "452f7632a6619848df887dc92522e81c97a0d9dc26b7a6c01a55e2b36c88ae1c",
    "114823/unprocessed/7T/Diffusion/114823_7T_DWI_dir72_AP.bvec":
        "1ea7c9d72f3444462ee35871dadc55314c639ddad2e206a0901df5e24fca744d",
}


def openneuro():
    by_release = {}
    for scan in SCANS:
        if scan.source != "HCP_1200":
            by_release.setdefault((scan.source, scan.version, scan.root), []).append(scan)
    for (accession, version, root), scans in by_release.items():
        clone(accession, version, root)
        files = []
        for scan in scans:
            for stem in scan.stems():
                files += [stem + ext for ext in (".nii.gz", ".bval", ".bvec")]
                json = root / (stem + ".json")
                if json.is_symlink() or json.exists():
                    files.append(stem + ".json")
        fetch(accession, version, root, files)


def hcp():
    files = [stem + ext for scan in SCANS if scan.source == "HCP_1200"
             for stem in scan.stems() for ext in (".nii.gz", ".bval", ".bvec")]
    for rel in files:
        dest = HCP / rel
        if not dest.exists():
            subprocess.run(["aws", "s3", "cp", "--only-show-errors", f"{HCP_BUCKET}/{rel}",
                            str(dest)], check=True)
        if checksum(dest) != HCP_SHA256[rel]:
            raise SystemExit(f"  sha256 mismatch: {rel}")
    print(f"  ok  {len(files)} files of HCP_1200 in {HCP}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-hcp", action="store_true", help="only the OpenNeuro releases")
    openneuro()
    if not ap.parse_args().skip_hcp:
        hcp()
