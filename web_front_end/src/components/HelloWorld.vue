<!--
 * @Autor: xujiahuan
 * @Date: 2020-05-11 13:46:02
 * @LastEditors: xujiahuan
 * @LastEditTime: 2020-05-11 17:54:20
 -->
<template>
  <div class="hello">
    <h1>{{ msg }}</h1>
    <textarea class="left_text" placeholder="请输入待识别文字"></textarea>
    <button @click="submit" class="s_button">分析</button>
    <div class="right_text">
      <div class="model">
        <button @click="BERT" class="model_button bert">BERT</button>
        <button @click="HMM" class="model_button">HMM</button>
        <button @click="CRF" class="model_button">CRF</button>
        <button @click="LSTM" class="model_button">LSTM-CRF</button>
      </div>
      <p>{{ ner_res}}</p>
    </div>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  data () {
    return {
      msg: '中文命名实体识别平台',
      ner_res: 'hello world',
      model: 'BERT'
    }
  },
  methods: {
    submit () {
      alert("dsdss");
    },
    BERT (){
      this.model = "BERT";
    },
    HMM (){
      this.model = "HMM";
    },
    CRF (){
      this.model = "CRF";
    },
    LSTM (){
      this.model = "LSTM"
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
.model {
  height:80px;
  width: 100%;
}
.model_button{
  margin-top: 10px;
  margin-left: 0px;
  padding: 10px 20px;
  background: gray;
  cursor: pointer;
}
.bert{
  background: palevioletred;
}
.s_button {
    float: left;
    margin-left: 50px;
    margin-top: 120px;
    background-color: #4CAF50;
    border: none;
    color: white;
    padding: 15px 32px;
    text-align: center;
    text-decoration: none;
    font-size: 16px;
    cursor: pointer;
}
h1, h2 {
  font-weight: normal;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
.left_text{
  float: left;
  margin-left: 296px;
  margin-top: 20px;
  padding-left: 10px;
  padding-top: 15px;
  border-radius: 5px;
  height: 400px;
  width: 350px;
  border:2px solid red;
}
.right_text{
  float: left;
  height: 415px;
  width: 400px;
  float: left;
  margin-top: 20px;
  margin-left: 50px;
  border:2px solid red;
  border-radius: 5px;
}
.right_text p{
  float: left;
  margin-left: 10px;
  margin-top: 15px;
}
</style>
