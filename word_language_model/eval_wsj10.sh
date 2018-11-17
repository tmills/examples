#!/bin/sh

# Arg1 model file to use
in_file=$1
seg_file=${in_file%%pt}segments
tree_file=${seg_file%%segments}parsed.linetrees.txt
nopunc_file=${tree_file%%txt}nopunc.txt

# Run the nn parser script to get F/J values for every position in wsj10:
echo "Parsing..."
python parse.py --cuda --model $in_file --input ~/Projects/db_cky/generated/wsj.noempty.linetrees.10.lower.linetoks > $seg_file

# Convert to trees:
echo "Converting to trees..."
cat $seg_file | python output2linetrees.py > $tree_file

# Convert to nopunc trees:
echo "Removing punctuation..."
python ~/Projects/db_cky/utils/delete_PU_nodes.py --style wsj --file $tree_file

# Run evalb
echo "Running evalb"
~/soft/EVALB/evalb -p ~/soft/EVALB/unlabeled.prm \
  ~/Projects/db_cky/generated/wsj.noempty.linetrees.10.lower.nopunc.linetrees \
  $nopunc_file | tail -27

