from typing import Dict
from datasets import load_dataset, DatasetDict, Image as HFImage

def load_image_text_dataset(name: str, image_col: str = "image", caption_field: str = "caption") -> DatasetDict:
    """
    Load Flickr8k or Flickr30k as a DatasetDict with columns:
      - 'image' (PIL)
      - 'captions' (List[str])
    Defaults preserved: name defaults to provided CLI value; signature kept.
    """
    name = name.lower()

    if name == "flickr8k":
        ds = load_dataset("jxie/flickr8k")
        def collect_caps(ex):
            return {"image": ex["image"], "captions": [ex[f"caption_{i}"] for i in range(5)]}
        ds = ds.map(collect_caps, remove_columns=[c for c in ds["train"].column_names if c != "image"])
        for split in ds.keys():
            ds[split] = ds[split].cast_column("image", HFImage())
        return ds

    if name == "flickr30k":
        # Load Parquet branch to avoid legacy loader script
        ds_all = load_dataset(
            "nlphuji/flickr30k",
            revision="refs/convert/parquet",
            split="test"
        )
        train = ds_all.filter(lambda e: e["split"] == "train")
        val   = ds_all.filter(lambda e: e["split"] == "val")
        test  = ds_all.filter(lambda e: e["split"] == "test")

        def collect_caps(ex):
            return {"image": ex["image"], "captions": ex["caption"]}

        keep_cols = ("image", "caption", "split")
        train = train.map(collect_caps, remove_columns=[c for c in train.column_names if c not in keep_cols])
        val   = val.map(collect_caps,   remove_columns=[c for c in val.column_names   if c not in keep_cols])
        test  = test.map(collect_caps,  remove_columns=[c for c in test.column_names  if c not in keep_cols])

        dd = DatasetDict(train=train, validation=val, test=test)
        for k in dd.keys():
            dd[k] = dd[k].cast_column("image", HFImage())
        return dd

    raise ValueError("Unknown dataset. Use 'flickr8k' or 'flickr30k'.")
