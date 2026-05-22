# P340 AI Intelligent Question-Answering Robot Case Study

## 1 Hardware Setup

Prepare a corkboard (the tape in the picture is A4 paper for positioning).

<img src="./img/0.png" width="40%" height="40%" />

Place the P340 robotic arm in the center of the bottom side of the corkboard.

<img src="./img/1.png" width="40%" height="40%" />

Reference distance between pen tip and pen clip: 22mm.

<img src="./img/10.png" width="60%" />

Build a camera stand and place the camera at the top to create a top-down view. The camera should be 330mm away from the wooden board. After powering on the robotic arm, connect the data cable and camera cable to the user's computer.

<img src="./img/3.png" width="40%" height="40%" />

## 2 Software Setup

**Notes**: Python version must be 3.8 or higher.

### Dependency Installation

```bash 
pip install -r requirements.txt
```
### API Key Acquisition: Users need to create their own Tongyi Qianwen and Deepseek accounts and configure their own API keys. API key usage is charged; users should check their Tongyi Qianwen and Deepseek account balances before use.

**Tongyi Qianwen API Key Application Reference Video**: https://www.bilibili.com/video/BV1iHVazvEvS/?spm_id_from=333.337.search-card.all.click&vd_source=672e3f7240eaaca210b45e7c033dc45f

**Deepseek** API Key Application Reference Video**: https://www.bilibili.com/video/BV1vCcge7E6W/?spm_id_from=333.337.search-card.all.click&vd_source=672e3f7240eaaca210b45e7c033dc45f

### Project Configuration 
Modify the config.yaml file in the /config directory to configure the software. Please ensure correct configuration before running the program. Enter your own Deepseek API key in the deepseek API key field, and enter your own Tongyi Qianwen API key in the qwen and qwen_vl API key fields. Enter the camera ID in the camera field. If using a laptop, the ID defaults to 1; if using a desktop computer, the ID is generally 0. Enter the actual serial port number of the robotic arm in the `com_port` field of the `robot` configuration file.

<img src="./img/8.png" width="40%" height="40%" />

### Project Operation
After ensuring the user's computer is connected to the network, run the `main.py` file, wait for the robotic arm to return to zero, and then tilt to one side.

<img src="./img/4.png" width="40%" height="40%" />

Fix the answer sheet in the center of the corkboard, with the bottom of the answer sheet firmly against the robotic arm.

<img src="./img/5.png" width="40%" height="40%" />

Enter 1 or 2 in the terminal and press Enter to confirm the mode. If you choose direct writing, enter the desired text in the terminal and press Enter. The robotic arm will then begin writing the user's input. If it's in AI answer mode, it will automatically open the camera to capture the content.

<img src="./img/9.png" width="40%" height="40%" />

Then manually adjust the camera and corkboard so that the image capture interface is as shown in the image below (leaving approximately one-sixth of the image blank). Be sure to adjust according to this image to ensure the camera view is as consistent as possible with the reference image below; otherwise, the writing area will be incorrect.

<img src="./img/6.png" width="70%"/>

 Press the spacebar to capture the image and wait for the rest of the process to complete, then obtain the final result.

<img src="./img/7.png" width="40%" height="40%" />