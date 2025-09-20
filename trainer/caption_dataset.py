
# The "train" python programs import this,
# to collect the filepaths of a "captioned dataset".
#  Actually loading in the dataset comes later

# Note that this is specifically targetted for use with the CACHING versions
# of my training scripts
# So it will go look for the img and text files as a safety....
# but it will only actually record the paths to the CACHE FILES
# See imgcache_suffix and txtcache_suffix

from pathlib import Path

from torch.utils.data import Dataset


class CaptionImgDataset(Dataset):
    """ Class to store paths to img/txt cache file files, as loaded from the
    one or more directory names passed at init.
    Both batch size and accum must be whole integers >= 1 
    They will then be used to cleanly truncate total number of files in the dataset, if needed.
    """
    def __init__(self, root_dirs, batch_size,
                 imgcache_suffix=".img_cache", txtcache_suffix=".txt_t5cache",
                 gradient_accum=1):
        self.files = []
        extset = ("jpg", "png")
        for root in root_dirs:
            print(f"Scanning {root} for {imgcache_suffix} and {txtcache_suffix} matching {extset}")
            subtotal=0
            for ext in extset:
                for p in Path(root).rglob(f"*.{ext}"):
                    img_cache = p.with_suffix(imgcache_suffix)
                    txt_cache = p.with_suffix(txtcache_suffix)
                    # Only keep samples where BOTH caches exist
                    if img_cache.exists() and txt_cache.exists():
                        self.files.append((img_cache, txt_cache))
                        subtotal+=1
            print(f"Cache pairs found: {subtotal}")

        if not self.files:
            raise RuntimeError("No valid cache pairs found! " \
                    "Did you run your cache pre-processing script?\n" \
                    f"{root_dirs}")

        print(f"Total cache pairs found: {len(self.files)}")

        num_batches = len(self.files) // batch_size
        even_batches = (num_batches // gradient_accum) * gradient_accum
        trimmed_len = even_batches * batch_size
        if trimmed_len < len(self.files):
            print(f"Trimming dataset from {len(self.files)} to {trimmed_len} for even accumulation.")
            self.files = self.files[:trimmed_len]

        print(f"Final dataset length: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_cache, txt_cache = self.files[idx]
        return {
            "img_cache": str(img_cache),
            "txt_cache": str(txt_cache),
        }

