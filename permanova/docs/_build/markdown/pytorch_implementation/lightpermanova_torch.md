# lightpermanova

### *class* permanova_torch.lightpermanova.LightPermAnova(sample_1: Tensor, compress: bool = True)

Bases: `object`

Class to perform [PERMANOVA](https://en.wikipedia.org/wiki/Permutational_analysis_of_variance)
analysis between two samples.

#### run_simulation(new_sample: Tensor, tot_permutations: int) → float

* **Parameters:**
  * **new_sample** (*torch.Tensor*) – sample that will be compared with sample_1
  * **tot_permutations** (*int*) – total number of permutations
* **Returns:**
  p_value (null hypothesis is sample_1 and new_sample are not different)
* **Return type:**
  float
