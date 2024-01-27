## Dive and underwater image and video color correction

**Sample images**

![Example](./examples/example.jpg)

This color correction script adjusts the red light levels in underwater images and videos.

  * ***Warning:*** Media files are altered *in-place*; backup orginal data before use!

### Setup
```
$ pip install -r requirements.txt
```

### Usage
```
./Color-Correct-Media.py "path/to/media"
🌊 Underwater Media Color Correction
🔍 Found:
📷   Images: 433
🎞    Videos: 14

✋ Do you want to continue? [Y]es/[N]o: Y
👌 Correcting color of:
    Image 𐎚 'P1150048.JPG'            9.0%|█▊                  | 39/433 [00:09<01:27,  4.49image/s]
```

### Share
If this repo was useful, please considering [sharing the word](https://twitter.com/intent/tweet?url=https://github.com/bornfree/dive-color-correction&text=Correct%20your%20dive%20footage%20with%20Python%20#scuba%20#gopro%20#python%20#opencv) on Twitter.

### Inspiration
This repo was inspired by the algorithm at https://github.com/nikolajbech/underwater-image-color-correction.
