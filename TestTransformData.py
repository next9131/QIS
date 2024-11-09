import ManageData as md
import TransformData as td
import numpy as np
SaveFile    = 'DataHighCap.pkl'
D           = md.loadproject(SaveFile)

TransformList  = ['ZSBT','ZSCS','ZSCSBS','ZSCSBI','FNBT', 
                  'FNCS','FNCSBS','FNCSBI',
                    'Q2CS','Q3CS','Q5CS','Q5CS','Q7CS','Q10CS']
#TransformList  = ['ZSBT']
#TransformList  = ['ZSCS']
#TransformList  = ['ZSCSBS']
#TransformList  = ['ZSCSBI']
#TransformList  = ['FNBT']
#TransformList  = ['FNCS']
#TransformList  = ['FNCSBY']
#TransformList  = ['FNCSBI']
#TransformList   = ['ZSBT']
X_AP_TD         = td.Transform(D['FCF_Price_AP'],TransformList,D)


