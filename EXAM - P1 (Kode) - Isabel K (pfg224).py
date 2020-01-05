#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@author: Isabel Kirkegaard (pfg224) - Open Data Science
"""

#Titanic.csv datasæt - Portfolio 1

#Opgave 1: Indholdet af datasættet
#a) Filen har forskellige datatyper som alder, køn og familienotater
# om passagerne. 
#Denne data blev først åbnet in text editor, for at se hvad datasættet
# indeholdt af datatyper. 
#Datasættet indeholder navne på passagerne, hvilken klasse de rejste med, 
#om de overlevede, deres køn og alder. 
#Dernæst indeholdt datasættet også prisen på deres billet, og om hvorvidt 
#de havde søskende, partnere eller forældre ombord. 

#b) Af data der mangler, så er der informationer der ikke vides noget om. 
#F.eks. er der et par personer, hvor der ikke vides om de havde familie
# ombord, eller hvad deres billet kostede. 


# ----------------


#Opgave 2a

#importerer pandas biblioteket i Python
import pandas as pd

#Der gøres brug af Pandas read_csv funktion for at læse csv-filen
data = pd.read_csv("titanic.csv")

#Viser hvor mange rækker der er
print(len(data))

#Viser hvor mange rækker og kolonner
print(data.shape)

#Fortæller hvad kolonnernes navne er
print(data.columns)

# Fortæller hvilke typer dataet er, f.eks. objects, int eller floats
print(data.dtypes)

#Viser at det er et dataframe
print(type(data))

#Opgave 3: Deskriptiv statistik
#Dette viser den deskriptive statistisk, altså hvad kan vi hive ud,
# der beskriver medianen, gennemsnittet osv.
print(data.describe())

#Hvor mange der overlevede
print(data['Survived'])

#Summen af de overlevende
print(data['Survived'].sum())

print(data['Age'].median())
#Viser medianen af alderen på passagerne



#-------------------------

#Opgave 4: Personer med samme efternavn?

#Der skabes en ny variabel last_names som fås ud fra titanic datasættets 'Name'. 
last_names = data['Name'].str.split(expand = True)
#Str.split returnerer en liste af seperate ord i en string. 
#Expand = true separerer string ind i kolonner, her 'Name'

#Value_counts tæller 'frequency counts' af elementer ved hjælp af Pandas
last_names[1].value_counts()

#'-1' peger på det sidste tegn i string ved 'Name', og dermed efternavne nu
last_names = (data['Name'].str.split().str[-1])

#Finder efternavne der går igen
print(last_names.value_counts())
#F.eks. så der 9 passagerer med efternavnet Andersson.


#------------------------

#Opgave 5: Pivot-tabel af de rejsendes klasser. 
#Tabellen viser at flest mennesker på første klasse overlevede

#Importerer numpy
import numpy as np 

#Med Pandas laver der en pivot-tabel hvor columns er klasse, og values er 'Age'
class_titanic = data.pivot_table(columns='Pclass', values='Age', aggfunc='count')

print(class_titanic)

#Pivot-tabel over hvilke klasser der havde flest omkomne. 
lost_lives_titanic = pd.pivot_table(data,index="Pclass", columns='Survived', values='Name', aggfunc='count')

#Det var 3. klasse med 368 tabte liv.
print(lost_lives_titanic)

