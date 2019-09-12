with open('urls.txt') as infile:
    contents = infile.read()
##for i in range(len(contents)):
##    if contents[i]=="," and contents[i+1:i+6]=="https":
##        contents.replace(contents[i],"\n")

print(contents)
