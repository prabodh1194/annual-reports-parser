## works on ec2-instance -- g6e.xlarge

aws ec2 create-security-group --group-name "qwen-embeddings" --description "for qwen embeddings" --vpc-id "vpc-08bc8fb7fb5322f29"

aws ec2 authorize-security-group-ingress --group-name "qwen-embeddings" --ip-permissions '{"IpProtocol":"tcp","FromPort":8000,"ToPort":8000,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}' '{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}' '{"IpProtocol":"tcp","FromPort":443,"ToPort":443,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}'

aws ec2 run-instances --image-id "ami-084568db4383264d4" --instance-type "g6e.xlarge" --key-name "ec2-kp" \
--block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-0edbe0f6601b2861c","VolumeSize":80,"VolumeType":"gp3","Throughput":125}}' \
--network-interfaces '{"SubnetId":"subnet-0a88da9b6432efc1d","AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-0789e8e758087c5e4"]}' \
--tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"qwen-embeddings"}]}' \
--instance-market-options '{"MarketType":"spot"}' \
--metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
--private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":false,"EnableResourceNameDnsAAAARecord":false}' \
--count "1"

sudo apt install python3.12

sudo apt update
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall

curl -LsSf https://astral.sh/uv/install.sh | sh

sudo reboot

uv init --name qwen -p 3.12
uv add vllm transformers hf_transfer "huggingface-hub[cli]"
