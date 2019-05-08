# sudoku-solver
This stuff just resolves sudokus one finger in the nose

Hi, my name is Rémi Lux. I'm currently in my final year of Computer Vision Degree.
Welcome to my project

### What is it doing ?
My algorithm is using powerful IA to extract digits.
Then, it will simply regenerate your image with the solution


| Version | Original Picture      |  Algorithm Output |
:-------------------------:|:-------------------------:|:-------------------------:
1.3 | <img src="https://user-images.githubusercontent.com/39727257/57191967-b146e100-6f2b-11e9-993d-3f6e8dc8e246.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57191964-aa1fd300-6f2b-11e9-8a3e-9403b851b2dd.jpg" width="400"/>
1.2 | <img src="https://user-images.githubusercontent.com/39727257/57106087-d5889f00-6d2c-11e9-805f-350233fed9bc.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57106093-d91c2600-6d2c-11e9-95da-c739bb21131e.jpg" width="400"/>
1.1 | <img src="https://user-images.githubusercontent.com/39727257/57035511-89692c00-6c52-11e9-852f-34acd3ed28e4.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57035569-abfb4500-6c52-11e9-9dd2-c7ca6e954a3f.jpg" width="400"/>
1.0 | <img src="https://user-images.githubusercontent.com/39727257/56866566-da6eeb00-69da-11e9-80bf-0f5eb124dce4.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/56866569-ec508e00-69da-11e9-9949-baaf827d3f6e.jpg" width="400"/>

### Why is it unique ?
Try it, you will see

### Is it really Working ?
Yes, of course, and you

### Where can I find weights for the CNN :( ?
DO NOT WORRY BRO, I GOT YOUR BACK

### Actual solving logger - 6 Grid Madness
Load everything 	63.3% - 1.029s  
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

- v1.4 : Video Handling
- v1.3 : More flexible / New training, better CNN !
- v1.2 : Multiple grids baby !
- v1.1 : Use probabilistic Hough & detect grid better
- v1.0 : First version
