## works on ec2-instance -- g6e.xlarge

aws ec2 create-security-group --group-name "qwen-embeddings" --description "for qwen embeddings" --vpc-id "vpc-08bc8fb7fb5322f29"

aws ec2 authorize-security-group-ingress --group-name "qwen-embeddings" --ip-permissions '{"IpProtocol":"tcp","FromPort":8001,"ToPort":8001,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}' '{"IpProtocol":"tcp","FromPort":8000,"ToPort":8000,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}' '{"IpProtocol":"tcp","FromPort":22,"ToPort":22,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}' '{"IpProtocol":"tcp","FromPort":443,"ToPort":443,"IpRanges":[{"CidrIp":"0.0.0.0/0"}]}'

aws ec2 run-instances --image-id "ami-084568db4383264d4" --instance-type "g6e.xlarge" --key-name "ec2-kp" \
--instance-market-options '{"MarketType":"spot"}' \
--block-device-mappings '{"DeviceName":"/dev/sda1","Ebs":{"Encrypted":false,"DeleteOnTermination":true,"Iops":3000,"SnapshotId":"snap-0edbe0f6601b2861c","VolumeSize":80,"VolumeType":"gp3","Throughput":125}}' \
--tag-specifications '{"ResourceType":"instance","Tags":[{"Key":"Name","Value":"qwen-embeddings"}]}' \
--metadata-options '{"HttpEndpoint":"enabled","HttpPutResponseHopLimit":2,"HttpTokens":"required"}' \
--private-dns-name-options '{"HostnameType":"ip-name","EnableResourceNameDnsARecord":false,"EnableResourceNameDnsAAAARecord":false}' \
--count "1" \
--network-interfaces '{"AssociatePublicIpAddress":true, "DeviceIndex":0, "Groups": ["sg-0789e8e758087c5e4"], "SubnetId": "subnet-021383e66ab47a71a"}' # 1c
# --network-interfaces '{"AssociatePublicIpAddress":true, "DeviceIndex":0, "Groups": ["sg-0789e8e758087c5e4"], "SubnetId": "subnet-0a88da9b6432efc1d"}' # 1a
# --network-interfaces '{"AssociatePublicIpAddress":true,"DeviceIndex":0,"Groups":["sg-0789e8e758087c5e4"],"SubnetId":"subnet-0bbb49aafcfe0cc5d"}' # 1b

# Get the Instance ID of the newly launched instance
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Name,Values=qwen-embeddings" "Name=instance-state-name,Values=running" \
  --query "Reservations[0].Instances[0].InstanceId" --output text)

# Get the Public IP address
PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

ZONE_ID=$(aws route53 list-hosted-zones-by-name --dns-name "openn.ai." --query "HostedZones[0].Id" --output text | sed 's|/hostedzone/||')

# Create a DNS A record in Route53 for emb.openn.ai
aws route53 change-resource-record-sets --hosted-zone-id "$ZONE_ID" \
  --change-batch '{
    "Changes": [{
      "Action": "UPSERT",
      "ResourceRecordSet": {
        "Name": "emb.openn.ai.",
        "Type": "A",
        "TTL": 300,
        "ResourceRecords": [{"Value": "'"$PUBLIC_IP"'"}]
      }
    }]
  }'

sudo apt update
sudo apt install -y python3.12 ubuntu-drivers-common build-essential python3.12-dev

sudo ubuntu-drivers autoinstall

curl -LsSf https://astral.sh/uv/install.sh | sh

sudo reboot

uv init --name qwen -p 3.12
uv add vllm transformers hf_transfer "huggingface-hub[cli]"
