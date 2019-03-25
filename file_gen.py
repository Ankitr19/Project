import os
val = 20.67
text = "The value is "+str(val)
new_file = open("data.txt", "w")
new_file.write(text)
new_file.close()
