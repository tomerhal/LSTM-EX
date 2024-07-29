from inference_levels.local_inferencers.local_inference import BaseLocalInference
import tests.unit_tests.base_local_infrerence_tests as bli_tsts

hidden_size = 10
bli = BaseLocalInference(hidden_size)

bli_tsts.main(bli, hidden_size)
