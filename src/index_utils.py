from typing import Any, Dict, List, Tuple

def build_eval_index(hf_dataset) -> Tuple[list, list, List[List[int]], List[int]]:
    """
    Convert a split with columns:
      - image: PIL image
      - captions: List[str] (>=1)
    into:
      - images: list of images (N)
      - texts: list of all captions (M)
      - img_to_txt: list of lists; img_to_txt[i] -> indices of texts for image i
      - txt_to_img: list of ints; txt_to_img[j] -> image index for text j
    """
    images = []
    texts = []
    img_to_txt = []
    txt_to_img = []

    for i in range(len(hf_dataset)):
        ex = hf_dataset[i]
        images.append(ex["image"])
        caps = ex["captions"]
        cur_txt_ids = []
        for c in caps:
            cur_txt_ids.append(len(texts))
            texts.append(c)
            txt_to_img.append(i)
        img_to_txt.append(cur_txt_ids)

    return images, texts, img_to_txt, txt_to_img
