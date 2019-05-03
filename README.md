# sudoku-solver
This stuff just resolve sudokus one finger in the nose

Hi, my name is Rémi Lux. I'm currently in my final year of Computer Vision Degree.
Welcome to my project

### What is it doing ?
My algorithm is using powerful IA to extract digits.
Then, it will simply regenerate your image with the solution


| Version | Original Picture      |  Algorithm Output |
:-------------------------:|:-------------------------:|:-------------------------:
1.2 | <img src="https://user-images.githubusercontent.com/39727257/57106087-d5889f00-6d2c-11e9-805f-350233fed9bc.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57106093-d91c2600-6d2c-11e9-95da-c739bb21131e.jpg" width="400"/>
1.1 | <img src="https://user-images.githubusercontent.com/39727257/57035511-89692c00-6c52-11e9-852f-34acd3ed28e4.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/57035569-abfb4500-6c52-11e9-9dd2-c7ca6e954a3f.jpg" width="400"/>
1.0 | <img src="https://user-images.githubusercontent.com/39727257/56866566-da6eeb00-69da-11e9-80bf-0f5eb124dce4.jpg" width="400"/> | <img src="https://user-images.githubusercontent.com/39727257/56866569-ec508e00-69da-11e9-9949-baaf827d3f6e.jpg" width="400"/>

### Why is it unique ?
Try it, you will see

### Is it really Working ?
Yes, of course, and you

### Where can I find weights for the CNN :( ?
DO NOT WORRY BRO, I GOT YOUR BACK

### Actual solving logger
Load everything 	59.0% - 0.651s  
Grid Research 		10.5% - 0.116s  
Digits Extraction 	5.7% - 0.063s  
Grid Solving 		23.4% - 0.259s  
Image recreation 	1.4% - 0.015s  
EVERYTHING DONE 	1.10s

--> Except the importing phase (on which I am powerless), 
I will try to optimise each part of the process starting by the grid solving
### TODO LIST
- [ ] More Robust
- [ ] Faster
- [X] Better
- [ ] Stronger
- [ ] Video Live Solving

### VERSION LIST

- v1.2 : Multiple grids baby !
- v1.1 : Use probabilistic Hough & detect grid better
- v1.0 : First version


[sudo_double]: https://user-images.githubusercontent.com/39727257/57106087-d5889f00-6d2c-11e9-805f-350233fed9bc.jpg
[sudo_double_solved]: https://user-images.githubusercontent.com/39727257/57106093-d91c2600-6d2c-11e9-95da-c739bb21131e.jpg
[imgSudoku0NotFilled]: https://user-images.githubusercontent.com/39727257/56866566-da6eeb00-69da-11e9-80bf-0f5eb124dce4.jpg
[imgSudoku0Solved]: https://user-images.githubusercontent.com/39727257/56866569-ec508e00-69da-11e9-9949-baaf827d3f6e.jpg
[imgSudoku1NotFilled]: https://user-images.githubusercontent.com/39727257/57035511-89692c00-6c52-11e9-852f-34acd3ed28e4.jpg
[imgSudoku1Solved]:https://user-images.githubusercontent.com/39727257/57035569-abfb4500-6c52-11e9-9dd2-c7ca6e954a3f.jpg
