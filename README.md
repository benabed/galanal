# galanal
Bunch of code to measure galactic contribution in CMB maps

**Version October 2022**
Correspond to the fits for the SPT3G winter fields.
Data is in `data/rq_spt_winter.cldf` and can be read with `galanal.rq`. It corresponds to all the TT, TE and EE cross spectra between 100 to 545 half mission maps.

`note_TE.py` is the more complete code to fit for the TE dust contribution. `note_EE.py` and the TT files are in lower state of polish...

The chains are all stored in `chains` and the final plots in `plots`. 

