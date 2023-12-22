import NextAuth from "next-auth/next";
import GoogleProvider from "next-auth/providers/google"
import { addToWhitelist } from "@/app/db";

export default NextAuth({
    providers: [
        GoogleProvider({
            //@ts-ignore
            clientId: process.env.GOOGLE_CLIENT_ID,
            //@ts-ignore
            clientSecret: process.env.GOOGLE_CLIENT_SECRET
        })
    ],
    callbacks: {
      async signIn({ user }) {
        if (user.email) {
            await addToWhitelist(user.email);
        }
        return true;
      },
    },
  });
