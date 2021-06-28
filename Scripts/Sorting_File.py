class file_sorting():

    def sorting(self, filename):
    
        infile = open(filename)
    
        x=[]
        y=[]
        z=[]
        index=[]
        radius=[]
    
        for line in infile:
        
            temp = line.split()
        
            x.append(temp[0])
            y.append(temp[1])
            z.append(temp[2])
            index.append(temp[3])
            radius.append(temp[4])
    
        infile.close()

        return x, y, z, index, radius


#x, y, z, index, radius = sorting("..\Data\lines.txt")
#print(z)