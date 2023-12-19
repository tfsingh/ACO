import NextAuth from "next-auth/next";
import GoogleProvider from "next-auth/providers/google"

export default NextAuth({
    providers: [
        GoogleProvider({
            //@ts-ignore
            clientId: process.env.GOOGLE_CLIENT_ID,
            //@ts-ignore
            clientSecret: process.env.GOOGLE_CLIENT_SECRET
        })
    ]
})