import urllib.request
import random


def downloader(image_url):
    file_name = random.randrange(1,10000)
    full_file_name = str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)


downloader('https://www.image-restore.co.uk/new/wp-content/uploads/water_damage_restored_blog.jpg')
