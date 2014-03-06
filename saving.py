import csv


def save_text(V,f):
    """
    Save text as a file 
    """
    V[ spikes ] = Vre
    for i in xrange(N):
        for j in xrange(N):
            f.write(str(V[i][j])+'_')
        f.write('\n') #Jump at the end of the column

def save_text_csv(V,mywriter):
    for row in V:
        mywriter.writerow(row)
        mywriter.writerow([])
   
  
