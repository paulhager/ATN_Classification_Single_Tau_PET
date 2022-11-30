# ATN Classification from Single Tau PET

Open source code for the paper "Full ATN Classification from Single Tau PET using Machine Learning"

To run the docker container and query the model: 
1. Install [docker](https://docs.docker.com/get-docker/)
2. From the terminal run `docker run -dit -p 5000:5000 paulhager2/atn:latest`. If you want to access the server on a different port locally, replace the first `5000` with your desired port
3. Open a browser and navigate to `localhost:5000` (or whichever port you choose)
4. Specify the age, sex, APOE genotype of the patient and upload their tau-PET volume
5. Accept the zip file download with the resulting regional and global predictions / extractions
