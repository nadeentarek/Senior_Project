const express = require("express");
const app = express();
const mongoose = require("mongoose");
const bodyParser = require("body-parser");
const ejs = require("ejs");
app.use(bodyParser.urlencoded({extended:true}));
app.set('view engine','ejs')
mongoose.connect("mongodb+srv://b00077846:adhampwd@anomalydetection.srxis.mongodb.net/myFirstDatabase?retryWrites=true&w=majority",{useNewUrlParser:true},{useUnifiedTopology: true})
const readingSchema = {
    pair_id: String,
    sensor1_id : String,
    sensor2_id : String,
    bunker_id : Number,
    anomaly: Boolean,
    timestamp : Date
}
const Reading = mongoose.model("new_readings",readingSchema);
var mqtt = require('mqtt')
var count =0;
var client = mqtt.connect("mqtt://broker.hivemq.com");
var topic = 'siloAnomaly/#'


    client.on('connect', ()=>{
        client.subscribe(topic)
        console.log("subscribed successfully to topic:",topic)
    })
    client.on('message', (topic, message)=>{
        
        message = message.toString()
        console.log("Message recieved!")
        
        var today = new Date();
        console.log("today is:",today);
        //today.setTime(today.getTime() +4*60*60*1000);
        //console.log("new today is, after adding 4 hours:",today);
        Anomaly = true
        let newReading = new Reading({
            pair_id: 7,
            sensor1_id : '70b3d50680000fba',
            sensor2_id : '70b3d50680000950',
            bunker_id : 1,  
            anomaly: Anomaly,
            timestamp : today
        });
        newReading.save();
        //alert("ANOMALY DETECTED!!!!")
    })
 



/*
const readingSchema = {
    pair_id: String,
    sensor1_id : String,
    sensor2_id : String,
    bunker_id : Number,
    reading_one : Number,
    reading_two : Number,
    anomaly : Boolean,
    timestamp : Date
}
const Reading = mongoose.model("readings",readingSchema);
/*
app.get("/",function(req,res){ 
    res.sendFile(__dirname +"/index.html")
})
*/
/*
app.get("/",async function(req,res){ 
    const numRecords =  await Reading.countDocuments({}).exec(); 
    const numAnomalies =  await Reading.countDocuments({anomaly:true}).exec();
    const numNormal = numRecords - numAnomalies;
    const maxDatePair = await Reading.find({}).sort([['timestamp', -1]]).where('anomaly').equals(true).limit(1).exec();
    const AllReadings = await Reading.find({}).exec(); 
    const anomPairs = await Reading.find({}).sort([['timestamp', -1]]).where('anomaly').equals(true).exec();
    console.log("ANOM PAIRS:");
    console.log(anomPairs);
    var pair_id;
    var time;
    var anompair_ids = [];
    const sensor_pair_temp = {};
    var index=0;
    maxDatePair.forEach(record =>
        {  
        pair_id = record.pair_id;
        const d = new Date(record.timestamp);
        console.log("new date is");
        console.log(d)
        let h = d.getHours();
        let m = d.getMinutes();
        let s = d.getSeconds();
        time = h + ":" + m + ":" + s;
        })

        anomPairs.forEach(record =>
            {  
             const sensorpair_id = record.pair_id;
             anompair_ids.push('pair '+sensorpair_id);
            })

        AllReadings.forEach(record =>
        {  
            const spair_id = record.pair_id;
            sensor_pair_temp['pair '+spair_id] = ((record.reading_one + record.reading_two)/2).toFixed(1);
            });
        console.log(sensor_pair_temp);
        const keys = Object.keys(sensor_pair_temp);
        const vals = Object.values(sensor_pair_temp);
        
          vals.forEach((element, index) => {
          vals[index] = parseFloat(element);
          console.log(vals[index])
            });
          
       //console.log(keys);
       // console.log(typeof keys[0])
        //console.log(vals);
        //console.log(typeof vals[0])
        console.log(sensor_pair_temp);
        console.log(anompair_ids);
        console.log(AllReadings)
        console.log('maxdatepair:',maxDatePair)
        res.render('index',{num_readings: numRecords ,num_anomalies :numAnomalies, num_pair:pair_id, num_time: time, num_normal : numNormal, num_keys: keys, num_vals : vals, lst_anomalies: anompair_ids});
});

*/

app.get("/",async function(req,res){ 
    const numRecords =  await Reading.countDocuments({}).exec(); 
    const numAnomalies =  await Reading.countDocuments({anomaly:true}).exec();
    const numNormal = numRecords - numAnomalies;
    const maxDatePair = await Reading.find({}).sort([['timestamp', -1]]).where('anomaly').equals(true).limit(1).exec();
    const AllReadings = await Reading.find({}).exec(); 
    const anomPairs = await Reading.find({}).sort([['timestamp', -1]]).where('anomaly').equals(true).exec();
    var pair_id;
    var time;
    var anompair_ids = [];
    const sensor_pair_temp = {};
    var index=0;
    console.log('mdp',maxDatePair)
    maxDatePair.forEach(record =>
        {  
        pair_id = record.pair_id;
        
        const d = new Date(record.timestamp);
        console.log("date",d)
        let h = d.getHours();
        console.log("h is",h);
        let m = d.getMinutes();
        let s = d.getSeconds();
        time = h + ":" + m + ":" + s;
        console.log(time)
        })

        anomPairs.forEach(record =>
            {  
             const sensorpair_id = record.pair_id;
             anompair_ids.push('pair '+sensorpair_id);
            })

        AllReadings.forEach(record =>
        {  
            const spair_id = record.pair_id;
            sensor_pair_temp['pair '+spair_id] = record.anomaly;
            });
        const keys = Object.keys(sensor_pair_temp);
        const vals = Object.values(sensor_pair_temp);
        
        new_pairs = [];  
        for (var i = 1; i <=44; i++) 
        {
            new_pairs.push('pair '+i);
        }
        time_string = []
        value_string =[]
        console.log(keys.length)
        AnomalyHistory = await Reading.find({}).where('pair_id').equals(7).exec();

        AnomalyHistory.forEach(record =>
            {  
                time_string.push(record.timestamp) 
                 if(record.anomaly)
                 {
                        value_string.push(0)
                 }
                 else
                 {
                        value_string.push(1)
                 }
                });
       
       for (i=0;i<time_string.length;i++)
       {
           time_string[i] = time_string[i].toLocaleTimeString();
       }    
      
        //console.log(sensor_pair_temp);
       // console.log(anompair_ids);
        //console.log(AllReadings)
       // console.log('maxdatepair:',maxDatePair)
        res.render('index',{num_readings: numRecords ,num_anomalies :numAnomalies, num_pair:pair_id, num_time: time, num_normal : numNormal, num_keys: keys, num_vals : vals, lst_anomalies: anompair_ids, time_string: time_string, value_string: value_string});

});


app.get("/AllReadings",function(req,res){ 
    Reading.find({}, function(err,readings){
        res.render('readings',{allreadings: readings});
    })
   
}); 

app.get("/addreading.html",function(req,res){ 
    res.sendFile(__dirname +"/addreading.html")
})

app.get("/silo.jpg",function(req,res){ 
    res.sendFile(__dirname +"/silo.jpg")
})

app.get("/silo2.jpg",function(req,res){ 
    res.sendFile(__dirname +"/silo2.jpg")
})

app.get("/silo1.jpeg",function(req,res){ 
    res.sendFile(__dirname +"/silo1.jpeg")
})

app.post("/",function(req,res){
    let newReading = new Reading({
        pair_id: req.body.pair_id,
        sensor1_id :  req.body.sensor1_id,
        sensor2_id : req.body.sensor2_id,
        bunker_id : req.body.bunker_id,
        reading_one : req.body.reading_one,
        reading_two : req.body.reading_two,
        anomaly :  (req.body.anomaly =='true'),
        timestamp : req.body.timestamp
    });
    newReading.save();
    res.redirect('/');
});

app.listen(3004,function () {
    console.log("server is running on 3004");
})