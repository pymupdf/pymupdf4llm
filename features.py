import os
import tempfile
import subprocess as sp

global features_exe

#features_exe = "windows\\x64\Debug\\features.exe"
features_exe = "features"

def pdf_features_extraction(filename, rect_list):
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp:
            tmp.write("PDF,page,x0,y0,x1,y1,class,score,order\n")
            n = 0
            for r in rect_list:
                line = f'{filename},0,{r[0]},{r[1]},{r[2]},{r[3]},0,0,{n}'
                n = n + 1
                tmp.write(line+'\n')

        command = features_exe + " "+path
        result = sp.run(command, text=True, capture_output=True, shell=True)
        if result.returncode:
            print('Command failed:\n'+command+'\n')
        lines = result.stdout.splitlines()
        feature_rect_list = []
        for line in lines[1:]:
            features = []
            for x in line.split(','):
                features.append(x)
            feature_rect_list.append(features)
    finally:
        os.remove(path)

    return feature_rect_list

pdf_features_extraction("PMC1064098_00008.pdf", [[10, 10, 100, 100], [20, 20, 100, 100]])
