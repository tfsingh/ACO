import NextAuth from "next-auth/next";
import GoogleProvider from "next-auth/providers/google"


export default NextAuth({
    providers: [
        GoogleProvider({
            clientId: "",
            clientSecret: ""
        })
    ]
})