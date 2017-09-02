#!/usr/bin/env bash
#Please refer to src/README.md for an explanation of DATA PREP Phase 0

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}Ctrl-C to terminate any time${NC}"
echo -e "Enter your sudo password on prompt"
echo -e "Enter the top level folder path where you want to convert the DOCX ('${BLUE}none${NC}' if not converting):"
read -p ":" docxFolder
echo -e "Enter the top level folder path where you want to convert the PDF ('${BLUE}none${NC}' if not converting):"
read -p ":" pdfFolder

if [ $docxFolder != "none" ]
then
    echo -e "${GREEN}This script will convert all docx in $docxFolder to text${NC}"
fi

if [ $pdfFolder != "none" ]
then
    echo -e "${GREEN}This script will convert all pdf in $pdfFolder to text${NC}"
fi

read -n1 -r -p "Press any key to continue..." key

if [ $docxFolder != "none" ]
then
    sudo apt-get install docx2txt
    find $docxFolder -name *.docx -exec docx2txt '{}' \;
fi

if [ $pdfFolder != "none" ]
then
    sudo apt-get install pdftotext
    find $pdfFolder -name *.pdf -exec pdftotext -layout '{}' \;
fi

echo "Conversion completed."