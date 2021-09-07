set -e

# download google drive url prefix
download_googleDrive_url="https://docs.google.com/uc?export=download&id="

# training file url
training_fileID="1EqhYCjC7XziHS_mgAXAmYXBJClb0ZNDy"
download_training_data_url=$download_googleDrive_url$training_fileID # url+fileID

# testing file url
testing_fileID="179AfIHJ5_oh45bQAGNK82EMf8emKyxka"
download_testing_data_url=$download_googleDrive_url$testing_fileID

# save data dir
data_dir="./data"
if ! [ -d $data_dir ] ; then
	mkdir $data_dir
fi


function download_file_from_google_drive {
	download_url=$1
	file_name=$2

	save_file_path="$data_dir/$file_name.zip"
	wget --no-check-certificate $1 -O $save_file_path 
	unzip $save_file_path -d $data_dir
	rm $save_file_path
}


download_file_from_google_drive "$download_training_data_url" "train_data"
download_file_from_google_drive "$download_testing_data_url" "test_data"
