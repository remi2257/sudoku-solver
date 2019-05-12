# sudoku-solver
This stuff just resolves sudokus one finger in the nose

Hi, my name is Rémi Lux. I'm currently in my final year of Computer Vision Degree
in Grenoble INP - Phelma.
Welcome to my project

### What is it doing ?
For a single image process, my algorithm is using working as following :
- Preprocess image to enhance igh frequencies
- Find lines thanks to Hough Space
- Analyse lines to know which can correspond to a grid
- Extract grid and its digits thanks to my CNN
- Resolve the grid
- Then, it will simply regenerate your image with the solution

<p align="center">
<img src="https://user-images.githubusercontent.com/39727257/57585323-031cd780-74e7-11e9-9268-165ed31e8f7f.gif" width="640"/>
</p>


### Evolution of the Algorithm

| Version | Original Picture      |  Algorithm Output |
:-------------------------:|:-------------------------:|:-------------------------:
1.5 | Video | <img src="https://user-images.githubusercontent.com/39727257/57585124-0367a380-74e4-11e9-9f7e-e567b08a2fab.gif" width="400"/>
1.3 | <img src="https://user-images.githubusercontent.com/39727257/57191967-b146e100-6f2b-11e9-993d-3f6e8dc8e246.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57191964-aa1fd300-6f2b-11e9-8a3e-9403b851b2dd.jpg" width="400"/>
1.2 | <img src="https://user-images.githubusercontent.com/39727257/57106087-d5889f00-6d2c-11e9-805f-350233fed9bc.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57106093-d91c2600-6d2c-11e9-95da-c739bb21131e.jpg" width="400"/>
1.1 | <img src="https://user-images.githubusercontent.com/39727257/57035511-89692c00-6c52-11e9-852f-34acd3ed28e4.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57035569-abfb4500-6c52-11e9-9dd2-c7ca6e954a3f.jpg" width="400"/>
1.0 | <img src="https://user-images.githubusercontent.com/39727257/56866566-da6eeb00-69da-11e9-80bf-0f5eb124dce4.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/56866569-ec508e00-69da-11e9-9949-baaf827d3f6e.jpg" width="400"/>

### Why is it unique ?
Try it, you will see

### Is it really Working ?
Yes, of course, and you

### Where can I find weights for the CNN :( ?
I trained mine personally with on a dataset of 10k numeric digits
with data augmentation. It gave me a precision of 99.5+ %

### Actual solving logger - 6 Grid Madness
Load everything 	63.3% - 1.029s  (Libraries + Keras Weights)
Grid Research 		15.8% - 0.257s  
Digits Extraction 	14.4% - 0.234s  
Grid Solving 		3.6% - 0.058s  
Image recreation 	3.0% - 0.049s  
EVERYTHING DONE 	1.63s  

--> Except the importing phase (on which I am powerless), 
I will take a look on multi-threading to improve the execution time

### TODO LIST

- [X] More Robust
- [ ] Faster
- [X] Better
- [ ] Stronger
- [ ] Video Live Solving
- [ ] Improve Training
- [ ] Multi-threading Processing

### VERSION LIST

- v1.5 : Stabilizing video resolution | 12/05/19
    - Multiple checking if the grid is well detect
    - Jump solving step if seems to be already solved !
- v1.4 : Video Handling | 08/05/19
- v1.3 : More flexible / New training, better CNN ! | 05/05/19
- v1.2 : Multiple grids baby ! | 02/05/19
- v1.1 : Use probabilistic Hough & detect grid better | 01/054/19
- v1.0 : First version | 28/04/19
