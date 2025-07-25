

This directory is for utilities used in conjuntion with datasets.
Most of them expect the data to have been download with downloaded with img2dataset.
So, "file1.jpg", "file1.txt", etc.

A "tag" is a word, or set of words, comma seperated.
So a txt file containing

    dog, a house, fine artwork

has 3 tags, not 5


* compare_imgdirs.py       - Given two directories, display images with same names side-by-side
* count_tags.py            - Show distribution frequency of all tags
* dataset2hex.py           - Organize image datasets into a standardized tree
* extracttxtfromjsonl.py   - Util to add md5 sum of an image to its .json file
* jsonupdatemd5.py         - Util to add md5 sum of an image to its .json file
* remove_tags.py           - Util to remove a named tag from multiple .txt files
* verify-download-size.py  - Compare image actual size, to width/height present in .json


* txtmergemoon.py   
If you have a large jsonl file which you used with dataset2img, and
you chose a specific field to become the caption, it will typically write out the caption to .txt,
and also include it in the matching .json file.
However, if you want to add any other field to the .json, you can use this util

