"""Delete images in data/ocr_output/**/images that aren't referenced by sibling .md files."""
from pathlib import Path
import re
import argparse

IMG_REF_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')
ROOT = Path(__file__).resolve().parent.parent / "data" / "ocr_output"


def referenced_images(md_path: Path) -> set[str]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    names = set()
    for m in IMG_REF_RE.finditer(text):
        url = m.group(1).split()[0].strip()
        if url.startswith(("http://", "https://")):
            continue
        names.add(Path(url).name)
    # also catch HTML <img src="...">
    for m in re.finditer(r'<img[^>]+src=["\']([^"\']+)["\']', text):
        names.add(Path(m.group(1)).name)
    return names


def clean_dir(md_dir: Path, dry_run: bool) -> tuple[int, int]:
    images_dir = md_dir / "images"
    if not images_dir.is_dir():
        return 0, 0
    referenced = set()
    for md in md_dir.glob("*.md"):
        referenced |= referenced_images(md)
    kept = removed = 0
    for img in images_dir.iterdir():
        if not img.is_file():
            continue
        if img.name in referenced:
            kept += 1
        else:
            removed += 1
            print(f"  remove: {img.relative_to(ROOT)}")
            if not dry_run:
                img.unlink()
    return kept, removed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list only, don't delete")
    args = ap.parse_args()

    total_kept = total_removed = 0
    for images_dir in ROOT.rglob("images"):
        if not images_dir.is_dir():
            continue
        md_dir = images_dir.parent
        kept, removed = clean_dir(md_dir, args.dry_run)
        if kept or removed:
            print(f"[{md_dir.relative_to(ROOT)}] kept={kept} removed={removed}")
        total_kept += kept
        total_removed += removed

    action = "would remove" if args.dry_run else "removed"
    print(f"\nTotal: kept={total_kept}, {action}={total_removed}")


if __name__ == "__main__":
    main()
