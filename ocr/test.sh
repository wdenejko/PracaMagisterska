c=1
while [[ c > 25 ]] ; # Stop when file.txt has no more lines
do
    echo "Python script called $c times"
    python ocr.py # Uses file.txt and removes lines from it
    c=$(($c + 1))
done
