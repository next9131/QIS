import ManageData as md

Filename    = 'DataAllCap.xlsm'
SaveFile    = 'DataAllCap.pkl'
R1          = md.removeloadeddata(SaveFile)
D           = md.initialize()
D           = md.loaddataFormat1(D,Filename)
R2          = md.saveproject(D,SaveFile)
D           = md.loadproject(SaveFile)

Filename    = 'DataHighCap.xlsm'
SaveFile    = 'DataHighCap.pkl'
R1          = md.removeloadeddata(SaveFile)
D           = md.initialize()
D           = md.loaddataFormat1(D,Filename)
R2          = md.saveproject(D,SaveFile)
D           = md.loadproject(SaveFile)

Filename    = 'DataMidCap.xlsm'
SaveFile    = 'DataMidCap.pkl'
R1          = md.removeloadeddata(SaveFile)
D           = md.initialize()
D           = md.loaddataFormat1(D,Filename)
R2          = md.saveproject(D,SaveFile)
D           = md.loadproject(SaveFile)

Filename    = 'DataLowCap.xlsm'
SaveFile    = 'DataLowCap.pkl'
R1          = md.removeloadeddata(SaveFile)
D           = md.initialize()
D           = md.loaddataFormat1(D,Filename)
R2          = md.saveproject(D,SaveFile)
D           = md.loadproject(SaveFile)