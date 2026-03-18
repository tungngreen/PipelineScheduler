* inference's downstreams' `nb_expectedshape`s should be the same as post processor's `msvc_dataShape`
* Receiver's `msvc_dataShape` should be set during profiling
    * The list of `msvc_dataShape` is the list of potential shapes for profiling.
* postprocessor's downstreams' `nb_commMethod` should be matched with sender's downstreams' `nb_commMethod`
    * If sender communicates CPU data (`nb_commMethod = 2 or 3), it requires CPU requests from postprocessor.
    * Thus, in such a case, postprocessor's downstreams' `nb_commMethod` should be 3