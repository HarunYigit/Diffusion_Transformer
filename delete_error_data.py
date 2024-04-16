deletes = """noises\\1 (903)_noise_1.jpg
noises\\1 (903)_noise_2.jpg
noises\\1 (903)_noise_3.jpg
noises\\1 (903)_noise_4.jpg
noises\\1 (903)_noise_5.jpg
noises\\1 (903)_noise_6.jpg
noises\\1 (903)_noise_7.jpg
noises\\1 (903)_noise_8.jpg
noises\\1 (903)_noise_9.jpg
noises\\1 (903)_noise_10.jpg
noises\\1 (903)_noise_11.jpg
noises\\1 (903)_noise_12.jpg
noises\\1 (903)_noise_13.jpg
noises\\1 (903)_noise_14.jpg
noises\\1 (903)_noise_15.jpg
noises\\1 (903)_noise_16.jpg
noises\\1 (903)_noise_17.jpg
noises\\1 (903)_noise_18.jpg
noises\\1 (903)_noise_19.jpg
noises\\1 (903)_noise_20.jpg
noises\\1 (903)_noise_21.jpg
noises\\1 (903)_noise_22.jpg
noises\\1 (903)_noise_23.jpg
noises\\1 (903)_noise_24.jpg
noises\\1 (903)_noise_25.jpg
noises\\1 (903)_noise_26.jpg
noises\\1 (903)_noise_27.jpg
noises\\1 (903)_noise_28.jpg
noises\\1 (903)_noise_29.jpg
noises\\1 (903)_noise_30.jpg
noises\\1 (903)_noise_31.jpg
noises\\1 (903)_noise_32.jpg
noises\\1 (903)_noise_33.jpg
noises\\1 (903)_noise_34.jpg
noises\\1 (903)_noise_35.jpg"""

import os
for i in deletes.split("\n"):
    os.remove(i)
    print("dosya silindi")