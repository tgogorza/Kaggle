import pandas as pd

def clean_data(df):
    '''Cleanup data set (remove NaNs and useless columns)'''
    #Clean up the data set:
    #1. Convert columns to numeric values
    #2. Remove unwanted columns
    #3. Treat NAs (impute or remove) try fancyimpute
    
    ##Convert non-numeric columns to numeric values
    df['age'] = pd.to_numeric(df['age'], errors='coerce', downcast='integer')
    df['antiguedad'] = pd.to_numeric(df['antiguedad'], errors='coerce')
    #We can convert some columns to boolean (indresi, indext, sexo) 
    df.sexo = df.sexo.map(lambda x: 0 if x is 'V' else 1)
    df.indresi = df.indresi.map(lambda x: 0 if x is 'N' else 1)
    df.indext = df.indext.map(lambda x: 0 if x is 'N' else 1)
    #Map segmento, ind_empleado to values
    df.segmento = df.segmento.replace({ '01 - TOP':0, '02 - PARTICULARES':1, '03 - UNIVERSITARIO':2})
    df.ind_empleado = df.ind_empleado.replace({ 'N':0, 'F':1, 'B':2, 'A':3})
    #Map canal_entrada
    df = df[df.canal_entrada.notnull()]
    num_channels = len(df.canal_entrada.unique())
    channel_dict = {str(channel) : num for num, channel in enumerate(df.canal_entrada.unique()) }
    df.canal_entrada = df.canal_entrada.replace(channel_dict)
    
    ##Remove unwanted columns
    df = df.drop(['fecha_dato','ncodpers','fecha_alta','indfall','nomprov','tipodom','ult_fec_cli_1t'], axis=1)
    #indrel, inderel_1mes, tiprel and conyuemp show more than 99% of the same value, so no use for those columns
    df.indrel_1mes.value_counts()
    df.indrel.value_counts()
    df.tiprel_1mes.value_counts()
    df.conyuemp.value_counts()
    df = df.drop(['indrel','indrel_1mes', 'tiprel_1mes', 'conyuemp'], axis=1)
    #Countries other than spain represent less than 1%, so I'm just ignoring country column altogether
    spa = len(df[df.pais_residencia == 'ES'])
    notspa = len(df[df.pais_residencia != 'ES'])
    ratio = float(notspa) / (spa+notspa)
    df = df.drop(['pais_residencia'], axis=1)
    
    ##Remove NAs
    #Drop rows without country, age, segmento   (we could impute the values, but that could potientially introduce some bias)
    #df = df[df.pais_residencia.notnull()]
    df = df[df.age.notnull()]
    df = df[df.segmento.notnull()]
    #I consider rows with age under 18 to either have a wrong age value or to not be able to aquire new products, therefore I'll remove them  
    df = df[df.age >= 18]
    
    #Impute missing values for renta?
    #TODO
    
    return df
    