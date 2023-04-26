with open('pip.txt', 'r') as file:
    lines = file.readlines()

with open('new.txt', 'w') as file:
    for line in lines:
        new_line = line.split('==')[0] + '\n'
        file.write(new_line)