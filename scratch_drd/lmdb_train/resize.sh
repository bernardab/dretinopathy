for file in test/*; do convert -resize 10% $file test_resized/`basename $file`; echo $file; done
