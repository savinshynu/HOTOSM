"""
Script to download images from mapillary using an application token
Allows to dowload 60-70 images per minute
"""

import requests
import json
import pandas as pd
import mapillary.interface as mly

# mapillary token
mly_key = 'enter your token here'

# directory to save images and metdata
image_dir = "/Users/savin/Omdena-Projects/data-hotosm/images"
meta_dir = "/Users/savin/Omdena-Projects/data-hotosm"


# input coordinates. This one is collected from Ruiru city close to Nairobi in Kenya
# change the latitude and longitudes for your purpose
lon, lat = (36.96069030682938,
          -1.1501068311268483)

# check if the token is success
print(mly.set_access_token(mly_key))

# using mappillary library to collect the data from a given location. Radius is set to 2 km here.
data = mly.get_image_close_to(longitude=lon, latitude=lat, radius=2000).to_dict()

# Each regions contains several sequence of data collection and there are many images associated with each
# sequence_id
# dataframe to collect metadata of all possible sequences with in a region
metadata = pd.DataFrame(columns=['type', 'coordinates', 'captured_at', 'compass_angle', 'creator_id', 'id', 'is_pano', 'sequence_id'])

print(f"Number of sequences which has data :{len(data['features'])}")

# Iteratively unwrap the dictionary and collect only the required metadata
for i, dict in enumerate(data['features']):

    type = dict['geometry']['type']
    coordinates = dict['geometry']['coordinates']
    captured_at = dict['properties']['captured_at']
    compass_angle = dict['properties']['compass_angle']
    creator_id = dict['properties']['creator_id']
    id = dict['properties']['id']
    is_pano = dict['properties']['is_pano']
    sequence_id = dict['properties']['sequence_id']

    row = {'type': type, 'coordinates': coordinates, 'captured_at': captured_at, 'compass_angle': compass_angle, 'creator_id': creator_id, 
    'id': id, 'is_pano': is_pano, 'sequence_id': sequence_id}

    metadata.loc[i] = row


# Now we will at all images associated with a sequence_id and select some subset of images and add the corresponding image_id and existing information
# to a new dataframe
metadata_new = pd.DataFrame(columns=['type', 'coordinates', 'captured_at', 'compass_angle', 'creator_id', 'id', 'is_pano', 'sequence_id', 'image_id'])

n = 0 # Monitor how many images have been collected as every request may not be success
for i, seq in enumerate(metadata['sequence_id']):

    if n > 2000: # change this to collect more data
        print("Collected 2000 images")
        break

    # The URL for finding the image data
    url = f'https://graph.mapillary.com/image_ids?access_token={mly_key}&sequence_id={seq}'

    # sending the response       
    response = requests.get(url)
    # converting the slice of dataframe to a dictionary so that a new key can be added later
    meta = metadata.iloc[i,:].to_dict()

    image_ids = None # total number of images ids under a sequence

    if response.status_code == 200: # success
        json = response.json()
        image_ids = [obj['id'] for obj in json['data']]

        print(f"{len(image_ids)} of images under the sequence id: {seq}")

    if image_ids: # if list not empty

        # Rather than selecting all the samples. let's select every 20th image if that exist to reduce duplication and increase diversity
        image_ids_new = [image_ids[ind] for ind in range(0, len(image_ids), 20)]

        print(f"Selecting {len(image_ids_new)} images out of {len(image_ids)}")
        
        if image_ids_new: # if not empty
            for image_id in image_ids_new:
                # adding image_id to the dictionary and then adding that info to the new dataframe
                meta['image_id'] = image_id
                metadata_new.loc[n] = meta

                # Construct the API URL
                url = f"https://graph.mapillary.com/{image_id}?fields=id,thumb_1024_url&access_token={mly_key}"

                # Send request
                response = requests.get(url)
                data = response.json()

                # Get image URL
                image_url = data["thumb_1024_url"]

                # Download the image
                image_response = requests.get(image_url)
                with open(f"{image_dir}/{seq}-{image_id}.jpg", "wb") as file:
                    file.write(image_response.content)

                print(f"Downloaded image: {image_id}.jpg")

                n += 1

# save the metdata to csv file locally
metadata_new.to_csv(f"{meta_dir}/metadata.csv")