
entries = [entry.strip() for entry in open('database_images.txt', 'r')]

import random; random.shuffle(entries); random.shuffle(entries)

fo = open('database_images.txt.shuffle.txt', 'w')
for entry in entries:
  fo.write(entry + '\n')
fo.close()
  
