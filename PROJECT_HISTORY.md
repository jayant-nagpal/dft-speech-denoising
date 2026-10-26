# Project decisions

The project asks how a change from time samples to Fourier coordinates can help
reconstruct speech hidden by noise. Its development followed six questions:

1. **What does a Fourier coefficient mean?** Start with inner products and an
   orthogonal basis, then connect the matrix calculation to sampled audio and bins.
2. **How can the description follow a word?** Use overlapping short-time spectra
   and verify reconstruction before modifying the coefficients.
3. **Which components should be attenuated?** Estimate the background from an
   explicit calibration interval and use interpretable classical spectral gains.
4. **How should the method be evaluated?** Fix the dataset version, speaker groups,
   noise keys and score. Use validation for parameter selection, then examine
   test scores alongside spectral changes and speech-distortion diagnostics.
5. **Could simply turning the volume down explain the result?** The all-positive
   improvements prompted a supplementary validation-selected constant-gain
   control. The spectral method retained an advantage under the tested conditions.
6. **What remains unresolved?** Changing backgrounds, retained mixture phase,
   broader speaker coverage and perceptual evaluation determine future work.

The implementation keeps these decisions visible: a small numerical core,
separate experiment stages, explicit controls and traceable results. The report
uses the same sequence to connect the mathematics with the measured evidence.
