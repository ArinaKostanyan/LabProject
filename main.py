import image

hint = "Enter your path or type pass and default will be picture of Lena. \n \t"

count_of_path = 0
 
flag = True

sobel_kernel = None
gaussian_kernel = None

while flag:
    if count_of_path > 20:
        print("You've entered path name more than 20 times. Program will be stopped.")
        break
    count_of_path+=1
    
    path = input(hint)
    
    if path == "pass":
        print("Image of Lena is going to be operated.")
        path = 'imagefolder/lena.png'
        
    if not path.endswith(('.jpeg', '.jpg', '.png')):
        print("You've entered invalid path. \n")
    else:
        flag = False  
else:
    image.processing(path, sobel_kernel, gaussian_kernel)
