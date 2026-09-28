import csv
import re

def txt_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Première ligne = en-têtes (séparés par ;)
        headers = lines[0].split(';')
        writer.writerow(headers)
        
        # Lignes suivantes = données entre "" séparées par ;
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            
            # Extraire les valeurs entre guillemets et ignorer les \n dans les ""
            values = re.findall(r'"(.*?)"', line, re.DOTALL)
            
            # Nettoyer les retours à la ligne à l'intérieur des valeurs
            values = [v.replace('\n', ' ').replace('\r', '') for v in values]
            
            if values:
                writer.writerow(values)
    
    print(f"Conversion terminée : {output_file}")

# Utilisation
txt_to_csv('GPLA_Chalco.txt', 'resultat.csv')