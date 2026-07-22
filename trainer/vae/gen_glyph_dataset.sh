
# This generates a nice variety of dataset output, so that training
# picks up text-generalization truths rather than getting overly
# focused on a single font
#
OUTDIR="/data/models/text"

# The first one is actually the default font
for font in "Libertinus Serif" \
 "DejaVu Serif" "Noto Serif" "TeX Gyre Pagella" \
 "TeX Gyre Pagella" "Liberation Sans" "DejaVu Sans" "Noto Sans" \
 "Liberation Mono" "DejaVu Sans Mono"; do

	name=$(echo $font|sed 's/ /_/g')
	echo $name
	python gen_glyph_dataset.py --book pride_and_prejudice --font "$font" --out $OUTDIR/pride_14_$name --font-size 14
	python gen_glyph_dataset.py --book pride_and_prejudice --font "$font" --out $OUTDIR/pride_16_$name --font-size 16

done

