The Python script performs the necessary activity for deploying a PLS calibration model to a batch manufacturing production environment.
It targets the PAT application where a spectrometer measures unit products in a batch manufacturing. 
It performs the following action:
-Load the PLS model
-Accept user input of batch number and instrument ID 
-Search spectra in the designated folder
-Apply model to the spectra 
-calculate model diagnositics
-output model prediction in a SQL server database
-save and report model output in a PDF file

The script is also modified for deployment as a web app.
