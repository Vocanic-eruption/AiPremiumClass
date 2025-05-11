from tqdm import tqdm 

fixed = open('doubanbook_top250_comments_fixed.txt', 'w')
lines = [line for line in open('doubanbook_top250_comments.txt', 'r')]
for i, line in enumerate(tqdm(lines)):
    if i == 0: 
        fixed.write(line)
        prev_line = '' 
        continue 
    
    if line.split('\t')[0] == prev_line.split('\t')[0]: 
        if len(prev_line.split('\t')) == 6: 
            fixed.write(prev_line + '\n') 
            prev_line = line.strip() 
        else: 
            prev_line = "" 
    else: 
        if len(line.split('\t')) == 6:
            fixed.write(line) 
            prev_line = line.strip() 
        else: 
            prev_line += line.strip() 

fixed.close ()