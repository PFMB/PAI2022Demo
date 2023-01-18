# PAI Demos Repository

### Setup Recommended Steps: 
First create and activate an environment in conda. Run the following commands:

```bash
$ conda create -n pai python=3.7 jupyter
$ conda activate pai
```

Then, install the requirements. On the home directory run 
```bash
$ pip install -r requirements.txt
```

Start a jupyter notebook server.
```bash
$ jupyter notebook 
```

### Dockerfile (Thanks to Manuel Hässig)
You can also create a Docker container using:
```bash
$ docker build -t pai-demos . 
```

To run the docker container you can use:
```bash
$ docker run --network="bridge" -it --rm -p 8888:8888 pai-demos
```
Then open the link displayed on your terminal in your browser