import sys
import os
import hashlib
import struct
import subprocess
import collections


END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', ")"] # acceptable ways to end a sentence

train_sample = "./train_eli5"
val_sample = "./val_eli5"
test_sample = "./test_eli5"

finished_files_dir = "eli5"

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
# num_expected_cnn_stories = 92579
# num_expected_dm_stories = 219506


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      if (line!=""): lines.append(line.strip())
  return lines

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@answer" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  #source,target
  return lines[0], lines[3]


def write_to_bin(path_sample, out_prefix):
    
  with open(out_prefix + '.source', 'wt') as source_file, open(out_prefix + '.target', 'wt') as target_file:
    for idx,sample_file in enumerate(os.listdir(path_sample)):
      if idx % 1000 == 0:
        print("Writing story %i done" % (idx))
      # Get the strings to write to .bin file
      sample_path=os.path.join(path_sample,sample_file)
    
      source, target = get_art_abs(sample_path)

      # Write article and abstract to files
      source_file.write(source + '\n')
      target_file.write(target + '\n')

  print("Finished writing files")


if __name__ == '__main__':
  # Create some new directories
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Read the stories, do a little postprocessing then write to bin files
  write_to_bin(test_sample, os.path.join(finished_files_dir, "test"))
  write_to_bin(val_sample, os.path.join(finished_files_dir, "val"))
  write_to_bin(train_sample, os.path.join(finished_files_dir, "train"))