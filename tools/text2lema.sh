#! /bin/bash
pre=$1
cd /extra/scratch03/lcabello/Pictogramas/summaries;
for file in /extra/scratch03/lcabello/Pictogramas/summaries/cenicienta.lsa.txt; do 
filename=`basename $file`
echo $file
#cat $file | awk '{if($NF!=".") print $0," ."; else print $0}' | tr -s " "> $filename.tmp
cat $file | awk '{print "*.",$0}' > $filename.tmp
analyze -f es.cfg --nodate --noquant --noloc --rtk --noflush --nocomp < $filename.tmp |awk '{if($1!="*"){if($1) printf("%s ",$2);} else print ""  }' | awk '{for(i=2;i<=NF;i++) printf("%s ",$i); printf("\n") }' | /extra/ArchivoSonoro6/Language/Periodicos/textos/bin/num2textWacent.pl > $filename.lema
rm $filename.tmp
done
